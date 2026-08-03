"""Unit tests for the model core (model/prob_resid.py).

These test the pure, deterministic pieces of the Stuff+ model — feature derivation,
the contact-grid labelling, and the outcome run values — so they run in a second
without the trained artifacts or the 1.5 GB feature cache.
"""
import numpy as np
import pandas as pd
import pytest

import model.prob_resid as pr


def test_add_shape_features_arm_signing():
    """ax_arm / release_pos_x_arm flip sign for RHP and keep sign for LHP; az is raw."""
    df = pd.DataFrame({
        "p_throws": ["R", "L"],
        "ax": [10.0, 10.0],
        "az": [-15.0, -15.0],
        "release_pos_x": [2.0, 2.0],
    })
    pr.add_shape_features(df)
    # RHP: hs = -1  -> arm-signed columns negate. LHP: hs = +1 -> unchanged.
    assert df["ax_arm"].tolist() == [-10.0, 10.0]
    assert df["release_pos_x_arm"].tolist() == [-2.0, 2.0]
    # az is vertical — never handedness-signed.
    assert df["az"].tolist() == [-15.0, -15.0]


def test_add_shape_features_idempotent():
    df = pd.DataFrame({"p_throws": ["R"], "ax": [8.0], "az": [-12.0], "release_pos_x": [1.5]})
    pr.add_shape_features(df)
    once = df["ax_arm"].iloc[0]
    pr.add_shape_features(df)
    assert df["ax_arm"].iloc[0] == once


def test_handedness_mirror_gives_identical_shape():
    """A RHP and the mirror-image LHP (negated ax + release side) get identical
    arm-signed shape — the property that lets one scoring pass serve both hands."""
    rhp = pd.DataFrame({"p_throws": ["R"], "ax": [12.0], "az": [-10.0], "release_pos_x": [1.8]})
    lhp = pd.DataFrame({"p_throws": ["L"], "ax": [-12.0], "az": [-10.0], "release_pos_x": [-1.8]})
    pr.add_shape_features(rhp)
    pr.add_shape_features(lhp)
    for c in ("ax_arm", "az", "release_pos_x_arm"):
        assert rhp[c].iloc[0] == lhp[c].iloc[0]


def test_grid_cell_labels():
    """5-cell contact grid: 0 GB<95, 1 GB95+, 2 air<95, 3 air95+, 4 pop-up (grouped)."""
    df = pd.DataFrame({
        "launch_angle": [-5, 5, 20, 30, 60, 60],
        "launch_speed": [80, 100, 80, 100, 70, 105],
    })
    cell = pr._grid_cell(df)
    assert cell.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 4.0]  # pop-ups grouped, no EV split


def test_grid_cell_missing_ev_is_nan():
    """A ground/air ball with no exit velocity can't be placed in a hardness cell."""
    df = pd.DataFrame({"launch_angle": [5.0], "launch_speed": [np.nan]})
    assert pd.isna(pr._grid_cell(df).iloc[0])


def test_bucket_values_signs():
    """Whiffs and fouls both cost the batter run value; a whiff costs more than a foul."""
    n = 4000
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "description": ["swinging_strike"] * n + ["foul"] * n + ["ball"] * n,
        "delta_run_exp": np.r_[
            rng.normal(-0.10, 0.01, n), rng.normal(-0.04, 0.01, n), rng.normal(0.03, 0.01, n)
        ],
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
    assert pr._basefam("FC") is None  # cutters routed separately


def test_predict_global_rv_empty_frame():
    empty = pd.DataFrame({c: [] for c in pr.SHAPE_FEATS})
    out = pr.predict_global_rv(empty, {"swing": None, "grid": None})
    assert len(out) == 0
