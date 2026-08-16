"""Behavioral tests for scoring (model/predict.py + the trained three-group model).

The property tests load the trained ensemble_all.pkl and are skipped if the model has
not been trained yet, so the suite still passes on a fresh checkout. Run
`python main.py train` to enable them.
"""
import numpy as np
import pandas as pd
import pytest

import model.prob_resid as pr
from model.predict import display_grade
from model.submodels import load_ensemble

ENS = load_ensemble("all")
needs_model = pytest.mark.skipif(ENS is None, reason="model not trained (ensemble_all.pkl missing)")


def _raw(pitch_type="FF", **over):
    """A physically-valid raw pitch with shape + router features computed."""
    row = dict(p_throws="R", pitch_type=pitch_type, release_speed=95.0, release_spin_rate=2300.0,
               arm_angle=48.0, release_extension=6.5, release_pos_x=-1.8, release_pos_z=5.9,
               vx0=6.0, vy0=-138.0, vz0=-4.5, ax=-11.0, ay=27.0, az=-13.0, spin_axis=205.0)
    row.update(over)
    df = pd.DataFrame([row])
    pr.add_magnus(df); pr.add_shape_features(df)
    return df


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
def test_scores_from_kinematics():
    """A pitch scores from its trajectory alone — with OR without spin_axis (the induced
    features are kinematic), but is unscorable when a shape feature is missing entirely."""
    with_axis = pr.grade_pitches(_raw(), ENS)[0]
    without_axis = pr.grade_pitches(_raw(spin_axis=np.nan), ENS)[0]
    assert np.isfinite(with_axis) and np.isfinite(without_axis)
    assert with_axis == pytest.approx(without_axis)          # spin_axis does not affect the grade
    unscorable = pr.grade_pitches(_raw(release_speed=np.nan), ENS)[0]
    assert np.isnan(unscorable)                              # missing a shape feature -> no grade


@needs_model
def test_higher_velocity_grades_better():
    """Holding shape fixed, more velocity lowers xRV (a better pitch)."""
    slow = pr.predict_group_rv(_raw(release_speed=91.0), ENS)[0]
    fast = pr.predict_group_rv(_raw(release_speed=101.0), ENS)[0]
    assert fast < slow   # lower xRV = better


@needs_model
def test_all_groups_score():
    """Routing works end to end: a fastball (FB), a slider (BR), and a changeup (OFF) each
    grade to a finite number on the shared scale."""
    ff = pr.grade_pitches(_raw("FF"), ENS)[0]
    sl = pr.grade_pitches(_raw("SL", release_speed=86.0, release_spin_rate=2600.0,
                               ax=4.0, az=-32.0, spin_axis=120.0), ENS)[0]
    ch = pr.grade_pitches(_raw("CH", release_speed=86.0, ax=-15.0, az=-24.0, spin_axis=230.0), ENS)[0]
    assert np.isfinite(ff) and np.isfinite(sl) and np.isfinite(ch)


@needs_model
def test_grades_are_finite_and_bounded():
    rng = np.random.RandomState(3)
    df = pd.concat([_raw(release_speed=float(v)) for v in rng.uniform(88, 102, 20)],
                   ignore_index=True)
    g = pr.grade_pitches(df, ENS)
    assert np.isfinite(g).all()
    assert (g > 0).all() and (g < 200).all()
