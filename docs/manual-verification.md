# Manual verification

Verification of the Sprint 4 change: replacing the one global Stuff+ model with the
**fastball / non-fastball split** model (`v600_fbsplit_siera`). Manual checks are paired with
the automated suite in `tests/` — the automated tests pin invariants (spin_axis dependency,
line-drive exclusion, routing, velocity monotonicity), the manual checks confirm the grades
are sensible against the eye test.

## Environment

- Retrain: `python main.py train` on 2022–2025 (~2.9 M pitches, cached features).
- Board: per-(pitcher, pitch-type) grade of the average pitch, 2026 MLB, n ≥ 50.

## 1. Training completes and produces the expected model

**Expected:** one model artifact `ensemble_all.pkl` with two group models on separate scales;
the per-hand spin-axis offset saved; contact-score → run-value slopes negative (a
ground-ball-heavy shape lowers run value).

**Observed:** ✅
```
Cutter router: 14,461 FC->FB, 210,771 FC->BR of 225,232
values: whiff=-0.1130  foul=-0.0387
[FB]    RV=-0.0376*(GB-FB-PU)-0.0196  norm mean=-0.04549 std=0.00571  (swings 664,570, in-play 192,827)
[NONFB] RV=-0.0406*(GB-FB-PU)-0.0136  norm mean=-0.05386 std=0.00796  (swings 709,254, in-play 198,160)
Model version: 78768b705ae9
```
Both slopes are negative — a higher `GB − FB − PU` (more grounders, fewer fly balls) lowers
expected run value, as SIERA intends. spin_offset saved as `{R: 96.8°, L: 82.6°}`, matching
the documented convention offsets.

## 2. Fastballs are graded against fastballs (the point of the split)

**Expected:** velocity-plus-ride power arms lead the four-seams; Misiorowski and Miller at the
top; sinkers rate within the fastball group instead of being buried.

**Observed:** ✅
```
FF: Misiorowski 130 · Kempner 125 · Miller 124 · Lawrence 119 · Cuas 117 · Alvarado 113
SI: Holmes 124 · Garcia 117 · Loáisiga 116 · Cano 115 · Ginn 114
FC: Alvarado 129 · Roycroft 111 · Holmes 109
    Miller, Mason  FF 124   Misiorowski FF 130   Holmes, Clay SI 124 (SSW sinker rates elite)
```
Under the previous single-scale model these same fastballs graded ~104 (buried under breaking
balls). On the fastball scale they now separate correctly, and Holmes' seam-shifted sinker
rates elite among sinkers.

## 3. Separate per-group scales

**Expected:** each group centered near 100 on its own scale, so a fastball's grade is relative
to fastballs and a breaking ball's to breaking balls (not one shared scale).

**Observed:** ✅ the FB and NONFB norms above are distinct (`-0.0455/0.0057` vs
`-0.0539/0.0080`); breaking/offspeed no longer sit structurally above every fastball.

## 4. A card cannot be made until Statcast fills the 3D spin axis (also automated)

**Expected:** the Magnus/non-Magnus split needs `spin_axis`; a pitch with no spin axis is
unscorable (NaN grade) and gains a grade once the axis is filled in.

**Observed:** ✅ `tests/test_predict.py::test_spin_axis_required_to_score` passes — the same
pitch grades to a finite number with `spin_axis` present and to NaN without it. Confirmed
end-to-end: scoring the 2026 board, pitches missing `spin_axis` produce no grade.

## 5. Line drives are excluded from the contact head (also automated)

**Expected:** launch angles 10–25° (line drives) are dropped from the GB/FB/PU labelling —
line-drive rate is batter/luck, not a pitcher skill.

**Observed:** ✅ `tests/test_prob_resid.py::test_grid_cell_excludes_line_drives` passes; the
contact head trains only on ground balls, fly balls, and pop-ups.

## 6. Routing, handedness, velocity (automated)

**Observed:** ✅ `test_both_groups_score` (a fastball and a slider each grade finite on their
own scale), `test_release_side_is_arm_normalized`, and `test_higher_velocity_grades_better`
all pass — a 101 mph four-seam grades above the same shape at 91.

## Verification after refactoring

The change replaced the single-model scoring pass with two group models + per-group
normalization, and removed a dead back-compat block that wrote unused `norm_global*.pkl`
files. To confirm the refactor did not silently break scoring:

- **Import/interface check:** `predict_global_rv` / `predict_family_rv` kept as back-compat
  aliases; a repo-wide grep confirmed no remaining readers of the removed norm files.
- **End-to-end:** the production model loads and grades the 2026 board with finite grades for
  every pitch that has a spin axis; the spin_axis / group routing behave as above.
- **Automated suite:** 17/17 tests pass after the refactor.

## Discovered limitations (recorded, not fixed)

- **The compressed fastball group inflates soft-tossers.** Because within-group standardization
  amplifies small differences, low-velocity four-seams float up: Zuber grades ~112 and Festa's
  dead 92 mph four-seam ~106, higher than they should among fastballs, while Eury Pérez's
  genuinely good four-seam sits ~98. The single-scale model kept these near 100 but buried the
  good fastballs; the split fixes the burying at the cost of this inflation. Tunable via the
  contact-score weights or a monotonic velocity prior — recorded as a known property of the
  shipped split model, not a regression.
