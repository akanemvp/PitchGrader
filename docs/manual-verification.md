# Manual verification

Verification of the Sprint 5 model (`v700_threegroup_gbair`): a three-group (fastball /
breaking / offspeed) probability model on **one shared scale** with **global run values** and a
**GB/air contact head** (exit-velocity heads removed). Manual checks are paired with the
automated suite in `tests/` — the automated tests pin invariants (feature derivation,
kinematics-only scoring, GB/air labelling, routing, velocity monotonicity), the manual checks
confirm the grades are sensible against the eye test.

## Environment

- Retrain: `python main.py train` on 2022–2025 (~2.9 M pitches, cached features).
- Board: per-(pitcher, pitch-type) grade of the average pitch, 2026 MLB, scored out-of-sample.

## 1. Training completes and produces the expected model

**Expected:** one model artifact `ensemble_all.pkl` with three group models on one shared
scale; one global set of outcome run values; a negative ground-ball value and positive air
value.

**Observed:** ✅
```
global values: Vwh=-0.1129 Vfo=-0.0387 Vgb=-0.0452 Vair=+0.1291
[FB]  (8f) outcome n=664,570 (rounds=211), in-play n=248,236 (GB/air classifier rounds=83)
[BR]  (8f) outcome n=514,259 (rounds=558), in-play n=179,990 (GB/air classifier rounds=155)
[OFF] (8f) outcome n=194,995 (rounds=199), in-play n=72,739  (GB/air classifier rounds=44)
Model version: 0ac2c07911b8
```
A whiff costs the batter the most (−0.113), a foul less (−0.039), a ground ball is mildly good
for the pitcher (−0.045), and an air ball is bad (+0.129) — the expected signs. All three
groups share the one norm; values are global.

## 2. Grades match the eye test

**Expected:** velocity-plus-ride power arms lead the four-seams; elite breaking balls top their
groups; a dead-average pitch sits near 100; no blow-ups.

**Observed (2026 board):** ✅
```
FF: Misiorowski 118 · Miller 114 · Scott 110 · Alvarado 110
SI: Cano 111 · Little 110 · Holmes 106
FC: Alvarado 124 · Roycroft 115 · Misiorowski 113
CU: Jax 135 · Taylor 132 · Glasnow 131 · Misiorowski 131
per-type mean: KC 113 · FS 111 · CH 108 · CU 108 · SL 107 · ST 102 · FC 99 · SI 95 · FF 94
```
Misiorowski and Miller lead the four-seams; Alvarado's cutter (routed to the breaking group)
and the sharp curveballs sit at the top; Skenes' four-seam (95) and the field spread sensibly.
No pitch runs to an extreme outlier.

## 3. One shared scale + global values (not per-group)

**Expected:** all three groups z-scored on a single scale, so fastballs sit below breaking
balls on average (breaking balls miss more bats and induce softer contact) rather than each
group being re-centered to 100.

**Observed:** ✅ per-type means range from FF/SI ≈ 94–95 up to KC/FS ≈ 111–113 on the one
scale — fastballs are graded against the whole population, and the global values mean a whiff
is worth the same in every group.

## 4. A pitch scores from kinematics — `spin_axis` is not required (also automated)

**Expected:** the induced-movement features come from the pitch's trajectory, not the measured
spin axis, so a feed without `spin_axis` (minor leagues) still scores.

**Observed:** ✅ the same four-seam grades to **94.6 with `spin_axis` present and 94.6 with it
NaN** — identical. `tests/test_predict.py::test_scores_from_kinematics` pins this, and also
that a pitch missing a real shape feature (e.g. `release_speed`) is left ungraded (NaN).

## 5. The GB/air contact head — why exit velocity was removed

**Expected:** a simple GB/air classifier captures the repeatable contact signal at least as
well as the more complex exit-velocity machinery it replaced.

**Observed (out-of-sample, 2026, per pitcher×pitch-type):** ✅
- Pitch **shape predicts exit velocity barely at all** — Pearson r = 0.18 (r² = 0.03); the
  model's predicted EV spanned ~2.4 mph against 15.3 mph of real spread.
- The **EV term added no ranking skill**: with ground-ball probability held fixed, the EV-only
  contact score correlated −0.05 with actual contact run value (noise).
- **P(GB) alone ranked contact damage as well or better** than the full EV model
  (Spearman ≈ 0.19–0.23 vs xwOBAcon / dRE), and the full contact score was 93% just P(GB).

So the exit-velocity heads were removed and the in-play term is `P(GB)·V_gb + P(air)·V_air`.
This is recorded as a deliberate, evidence-based simplification, not a regression.

## 6. Routing, handedness, velocity (automated)

**Observed:** ✅ `test_all_groups_score` (a fastball, slider, and changeup each grade finite on
the shared scale), `test_release_side_is_arm_normalized` and
`test_induced_horizontal_is_arm_normalized` (a mirror-image lefty and righty grade
identically), and `test_higher_velocity_grades_better` (a 101 mph four-seam out-grades the same
shape at 91) all pass.

## 7. Automated suite

**Observed:** ✅ `python -m pytest tests/` → **19 passed**. The suite was updated this sprint to
match the current model (8 induced features, GB/air labelling, FB/BR/OFF routing,
kinematics-only scoring) after the model core changed.

## Discovered limitations (recorded, not fixed)

- **Shape-only by design.** The model cannot see command, sequencing, or tunneling, so a
  pitcher whose value comes from location rather than raw stuff will grade lower than his
  results suggest — this is intentional (it is a Stuff+ metric) but worth stating.
- **Fastballs sit below breaking balls on the shared scale.** Because every pitch is on one
  scale, four-seams average ~94 and breaking balls ~107. This reflects real run value (breaking
  balls miss more bats) but means a "100" is not the same achievement for a fastball as for a
  slider. A per-group scale was tried and rejected — the shared scale keeps grades comparable
  across types, which the product wants.
