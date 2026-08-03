# Manual verification

Verification of the Sprint 4 change: replacing the three-family Stuff+ model with the one
global swing-outcome model (`v500_global_swing_grid`). Manual checks are paired with the
automated suite in `tests/` — the automated tests pin invariants (arm-signing, monotonicity,
handedness), the manual checks confirm the grades are sensible against the eye test.

## Environment

- Retrain: `python main.py train` on 2022–2025 (~2.9 M pitches).
- Re-score: `python main.py score` (all live-season tables).
- Board: per-(pitcher, pitch-type) grade of the average pitch, 2026 MLB.

## 1. Training completes and produces the expected model

**Expected:** one model artifact `ensemble_all.pkl`; the three family artifacts removed;
sensible run values and grid cell values.

**Observed:** ✅
```
values: whiff=-0.1130  foul=-0.0387
grid RV=[GB<95=-0.106, GB95+=+0.052, air<95=+0.016, air95+=+0.425, pop=-0.237]
saved -> ensemble_all.pkl ; removed ensemble_{fb,os,br}.pkl
```
Whiffs and pop-ups help the pitcher (negative RV); hard air balls (air95+) are the most
damaging cell (+0.425). Matches intuition.

## 2. Fastball leaderboard matches the eye test

**Expected:** velocity-plus-ride power arms lead the four-seams; Misiorowski and Miller at the
top; no soft-tosser inflated to the top.

**Observed:** ✅
```
FF: Misiorowski 117 · Miller 110 · Scott 110 · Helsley 109 · Cease 108 · Zeferjahn 108 · Burns 107
    Miller, Mason  FF 110  (101.4 mph)
    Misiorowski    FF 117  (100.6 mph)
```

## 3. Grades are on one global scale (fastballs below breaking balls)

**Expected:** by design, one shared scale — breaking balls average above fastballs.

**Observed:** ✅ per-type mean grade `FF:93  SI:91  SL:108  ST:109  CH:107  FS:109`. This is the
intended behavior of a single-scale model (documented in
[global-swing-model.md](global-swing-model.md)); it is *not* a defect. Callers that want each
type centered at 100 use the retained per-family tagging.

## 4. Handedness symmetry (also automated)

**Expected:** a lefty and a righty throwing physically mirrored pitches grade identically.

**Observed:** ✅ `tests/test_predict.py::test_handedness_mirror_scores_identically` passes; the
arm-signed features are byte-identical for mirrored raw kinematics.

## 5. Velocity monotonicity (also automated)

**Expected:** holding shape fixed, more velocity → more whiffs → a better grade.

**Observed:** ✅ `test_whiff_probability_rises_with_velocity` and `test_higher_velocity_grades_better`
pass; a 101 mph four-seam grades above the same shape at 91.

## Verification after refactoring

The change refactored three code paths that used to loop over three family models
(`predict`, `_composite_pitch_grade`, and the `train` family loop) down to a single global
pass. To confirm the refactor did not silently break scoring:

- **Import/interface check:** `predict_family_rv` was kept as a back-compat alias so existing
  callers still resolve; a repo-wide grep confirmed no remaining references to the removed
  `.ensembles` / `_fam_norm` / per-family artifacts outside the experiments folder.
- **End-to-end re-score:** all eight live-season tables re-scored cleanly (exit 0), producing
  finite grades — e.g. `pitches_2026_scored` 461,325 rows with no NaN blow-ups.
- **Automated suite:** 15/15 tests pass after the refactor.

## Discovered limitations (recorded, not fixed)

- **Contact grid over-credits weak contact on four-seams.** Matt Festa's dead 92 mph four-seam
  grades 106 — slightly above Eury Pérez's 98 mph four-seam (105) — because the contact grid
  reads Festa's soft-contact shape favorably. This is a known property of the contact-grid
  valuation (the grid assumes hittable-looking shapes yield the contact its cells imply), and
  the same mechanism under-grades running power four-seams. It is documented here as a
  limitation of the shipped model rather than a regression; the whiff head alone ranks these
  pitches correctly. A future iteration (whiff + HR-only in-play for fastballs) would address
  it.
