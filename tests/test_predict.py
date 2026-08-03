"""Behavioral tests for scoring (model/predict.py + the trained global model).

The property tests (display grade, monotonicity, handedness) load the trained
ensemble_all.pkl and are skipped if the model has not been trained yet, so the
suite still passes on a fresh checkout. Run `python main.py train` to enable them.
"""
import numpy as np
import pandas as pd
import pytest

import model.prob_resid as pr
from model.predict import display_grade
from model.submodels import load_ensemble

ENS = load_ensemble("all")
needs_model = pytest.mark.skipif(ENS is None, reason="model not trained (ensemble_all.pkl missing)")

# A representative four-seam shape, in SHAPE_FEATS order.
#            velo   spin  ax_arm   az  arm_ang  relx_arm  relz   ext
_FF = dict(release_speed=95.0, release_spin_rate=2300.0, ax_arm=8.0, az=-12.0,
           arm_angle=45.0, release_pos_x_arm=-1.5, release_pos_z=5.8, release_extension=6.5)


def _shape_frame(**overrides):
    row = dict(_FF); row.update(overrides)
    return pd.DataFrame([row])[pr.SHAPE_FEATS]


def test_display_grade_below_knee_unchanged():
    sp = np.array([80.0, 100.0, 120.0, 125.0])
    assert np.allclose(display_grade(sp), sp)


def test_display_grade_soft_caps_above_knee():
    """Above the 125 knee, grades compress toward the 135 ceiling and stay monotone."""
    out = display_grade(np.array([130.0, 150.0, 300.0]))
    assert (out < 135.0).all()          # never exceeds the ceiling
    assert out[0] < out[1] < out[2]     # still monotonic
    assert out[0] > 125.0               # but above the knee


@needs_model
def test_whiff_probability_rises_with_velocity():
    """The swing head is monotone in velocity: a faster fastball misses more bats."""
    slow = _shape_frame(release_speed=91.0).apply(pd.to_numeric).values
    fast = _shape_frame(release_speed=101.0).apply(pd.to_numeric).values
    p_slow = ENS["swing"].predict_proba(slow)[0, 0]   # class 0 = whiff
    p_fast = ENS["swing"].predict_proba(fast)[0, 0]
    assert p_fast > p_slow


@needs_model
def test_higher_velocity_grades_better():
    """Holding shape fixed, more velocity lowers xRV (a better pitch)."""
    slow = pr.predict_global_rv(_shape_frame(release_speed=91.0), ENS)[0]
    fast = pr.predict_global_rv(_shape_frame(release_speed=101.0), ENS)[0]
    assert fast < slow   # lower xRV = better


@needs_model
def test_handedness_mirror_scores_identically():
    """Raw mirrored kinematics (RHP vs LHP) must produce the same xRV after arm-signing."""
    rhp = pd.DataFrame([{"p_throws": "R", "ax": -8.0, "az": -12.0, "release_pos_x": 1.5,
                         "release_speed": 95.0, "release_spin_rate": 2300.0, "arm_angle": 45.0,
                         "release_pos_z": 5.8, "release_extension": 6.5}])
    lhp = pd.DataFrame([{"p_throws": "L", "ax": 8.0, "az": -12.0, "release_pos_x": -1.5,
                         "release_speed": 95.0, "release_spin_rate": 2300.0, "arm_angle": 45.0,
                         "release_pos_z": 5.8, "release_extension": 6.5}])
    pr.add_shape_features(rhp); pr.add_shape_features(lhp)
    r = pr.predict_global_rv(rhp, ENS)[0]
    l = pr.predict_global_rv(lhp, ENS)[0]
    assert r == pytest.approx(l)


@needs_model
def test_elite_fastball_beats_dead_fastball():
    """Eye-test invariant: a 100 mph riding four-seam out-grades a flat 91."""
    gm, gs = ENS["norm"]["mean"], max(ENS["norm"]["std"], 1e-6)
    elite = pr.predict_global_rv(_shape_frame(release_speed=100.0, az=-9.0, release_spin_rate=2500.0), ENS)[0]
    dead = pr.predict_global_rv(_shape_frame(release_speed=91.0, az=-16.0, release_spin_rate=2050.0), ENS)[0]
    g_elite = 100 + (gm - elite) / gs * 10
    g_dead = 100 + (gm - dead) / gs * 10
    assert g_elite > g_dead


@needs_model
def test_predicted_grades_are_finite_and_bounded():
    rng = np.random.RandomState(3)
    df = _shape_frame().sample(1, random_state=1)
    df = pd.concat([_shape_frame(release_speed=v) for v in rng.uniform(80, 102, 25)], ignore_index=True)
    gm, gs = ENS["norm"]["mean"], max(ENS["norm"]["std"], 1e-6)
    g = 100 + (gm - pr.predict_global_rv(df, ENS)) / gs * 10
    assert np.isfinite(g).all()
    assert (g > 0).all() and (g < 200).all()
