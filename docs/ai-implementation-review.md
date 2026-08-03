# AI-assisted implementation review

The Sprint 4 model swap, its tests, and its docs were produced with an AI pair (Claude Code).
This records how AI was used for test generation and review, what was revised or rejected, and
how the AI-generated work was verified before it was trusted.

## AI-assisted test generation

The test suite in `tests/` was drafted with the AI from a plain-language spec of the model's
intended invariants:

- **Pure-function tests** (`test_prob_resid.py`) — feature arm-signing, the 5-cell contact-grid
  labelling, outcome run-value signs, and family mapping. These need no trained model, so they
  run in ~1 s and pin the deterministic core.
- **Behavioral tests** (`test_predict.py`) — velocity monotonicity, left/right handedness
  symmetry, the display-grade soft cap, and an eye-test invariant (a 100 mph riding four-seam
  out-grades a flat 91). These load the trained `ensemble_all.pkl` and `skipif` it is absent,
  so a fresh checkout still passes.

The AI was directed to test *properties* (monotonic, symmetric, bounded) rather than frozen
numeric outputs, so the suite survives a retrain without churny golden-value updates.

## AI-assisted test review

Each generated test was reviewed for whether it could actually fail:

- Tests asserting model *behavior* were run against the real trained artifact, not a mock, so a
  green result means the shipped model genuinely has the property.
- Value-sign tests use synthetic data with a known ground truth (e.g. whiffs drawn at −0.10,
  fouls at −0.04) so the assertion `V_whiff < V_foul < 0` is a real check, not a tautology.

## One generated test that was revised

The AI's first velocity test asserted monotonicity on the **composite `xRV`**. On review this
was too fragile: `xRV` folds in the heavily-smoothed contact grid, whose per-shape output is
deliberately noisy, so a strict `xRV(fast) < xRV(slow)` could flake on a single shape vector.
It was **split into two sharper tests**: `test_whiff_probability_rises_with_velocity` asserts
the clean, documented relationship directly on the swing head's `P(whiff)`, and
`test_higher_velocity_grades_better` keeps the end-to-end grade check on a wide velocity gap
(91 vs 101) where the signal dominates the grid noise. The revised pair is both stricter and
less flaky.

## One refactoring suggestion — accepted, and one — rejected

**Accepted:** when the three family models collapsed to one, the AI proposed keeping
`predict_family_rv` as a thin back-compat alias for the new `predict_global_rv` instead of
renaming every call site. Accepted — it kept the model swap inside the `model/` boundary and
avoided touching `app.py`, the scorer, and the profile builders. A repo-wide grep confirmed no
caller depended on the removed per-family internals.

**Rejected:** the AI noted that normalizing per pitch type (so every type centers at 100) would
make the buried fastball scale look more conventional. This was rejected because the
requirement was explicitly "one model, one scale"; per-type centering would have erased the
intended cross-type comparability and silently changed the deliverable. The single global scale
was kept and the trade-off documented instead.

## How AI-generated work was verified

- **Ran the suite:** `python -m pytest tests/` → 15 passed.
- **End-to-end re-score:** `python main.py train` then `score` on all live seasons completed
  with exit 0 and finite grades, confirming the refactor did not break inference.
- **Board sanity check:** the regenerated 2026 fastball leaderboard was compared against the
  eye test (Misiorowski / Miller on top) and against the prior model's board before accepting
  the numbers.
- **Static scan:** grepped the repo for references to the removed interface
  (`.ensembles`, `_fam_norm`, `ensemble_fb/os/br`) to prove nothing outside the experiments
  folder still relied on the old model shape.

No AI-generated code or test was committed until it had either passed the suite or been checked
against real scored data.
