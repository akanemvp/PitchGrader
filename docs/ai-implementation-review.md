# AI-assisted implementation review

The Sprint 4 model change (the fastball / non-fastball split, `v600_fbsplit_siera`), its tests,
and its docs were produced with an AI pair (Claude Code). This records how AI was used for
exploration, test generation, and review, what was revised or rejected, and how the
AI-generated work was verified before it was trusted.

## AI-assisted model exploration — the contact head

The hard design problem was the in-play (contact) head. The AI worked through several
candidates against a held-out 2026 eye-test board, and most were **rejected** for concrete,
diagnosable reasons:

- **Predict exit velocity → run value — rejected (inverted the model).** It graded elite
  velocity *worse*: harder pitches produce higher exit velo when hit ("faster in, faster out"),
  and only mistakes reach the in-play branch (selection), so the head learned "105 mph = easy
  homer" and buried the best fastballs.
- **Binary home-run head — rejected (too noisy).** Home runs are 4.5% of contact; with so few
  positives per leaf the linear-tree leaves extrapolated the sparse velocity tail wildly (a
  105 mph four-seam's grade collapsed). Regularizing it (large `min_data`) tamed the crash but
  left a residual velocity dip.
- **xwOBACON / barrel heads — worked but middling.** Both de-noised the HR signal, but still
  charged elite velocity for hard contact through the same confound.
- **SIERA-style GB/FB/PU with line drives excluded — accepted.** Modeling the *launch-angle
  mix* the pitcher actually controls (ground balls vs fly balls vs pop-ups) rather than contact
  *hardness* (which is mostly the batter) removed the velocity confound entirely and made the
  velocity curve monotonic. Excluding line drives (10–25°) is deliberate: line-drive rate is a
  batter/luck outcome, not a repeatable skill.

The value here was the AI running and *diagnosing* each dead end quickly, so the accepted
design was chosen because the others demonstrably failed, not on a hunch.

## AI-assisted test generation

The test suite in `tests/` was drafted from a plain-language spec of the model's invariants:

- **Pure-function tests** (`test_prob_resid.py`) — the Magnus/non-Magnus feature derivation and
  its spin_axis dependency, the SIERA GB/FB/PU labelling and its line-drive exclusion, outcome
  run-value signs, and FB/NONFB routing. No trained model needed, so they run in ~1 s.
- **Behavioral tests** (`test_predict.py`) — the spin_axis-required-to-score property, velocity
  monotonicity, both groups scoring, and the display-grade soft cap. These load the trained
  `ensemble_all.pkl` and `skipif` it is absent, so a fresh checkout still passes.

The AI was directed to test *properties* (monotone, bounded, NaN-when-unscorable) rather than
frozen numeric outputs, so the suite survives a retrain without golden-value churn.

## A generated behavior that was wrong — and caught by its own test

The first cut of `predict_group_rv` filled missing features with the column median
(`fillna(X.median())`), carried over from the old code. That silently **defeated the spin_axis
dependency**: a pitch with no spin axis got median-filled Magnus features and a bogus grade.
The `test_spin_axis_required_to_score` test failed on exactly this — a pitch with `spin_axis =
NaN` came back finite instead of NaN. The fix was to score only rows with all 10 features
present and return NaN otherwise, enforcing the dependency at the model level rather than
relying on the caller to pre-filter. The failing test is what surfaced the defect.

## Refactoring — accepted and rejected

**Accepted (cleanup):** the old `train.py` wrote four legacy `norm_global*.pkl` /
`norm_family*.pkl` files "for back-compat". A repo-wide grep found **no readers**, so the whole
block was deleted rather than fixed — keeping the model clean, as requested. `predict_global_rv`
/ `predict_family_rv` were kept as thin aliases so external callers still resolve.

**Rejected:** the AI also built the same model as a **single global model (no split)** to
compare. It was rejected because on one shared scale the good fastballs were buried (Miller 104,
Pérez 96) under breaking balls — the exact problem the split exists to solve. The single-model
run is kept only as the comparison that justifies the split; the split is the shipped model.

## How AI-generated work was verified

- **Ran the suite:** `python -m pytest tests/` → **17 passed**.
- **End-to-end retrain:** `python main.py train` completed with the expected per-group scales
  and saved every artifact (`ensemble_all.pkl`, `spin_offset.pkl`).
- **Board sanity check:** the 2026 board was compared against the eye test (Misiorowski / Miller
  lead the four-seams, Holmes' seam-shifted sinker rates elite, Alvarado's cutter routes to the
  breaking group) and against the scratchpad prototype's numbers before accepting.
- **Dependency check:** confirmed directly that a pitch with `spin_axis` scores and the same
  pitch without it does not.

No AI-generated code or test was committed until it had passed the suite and been checked
against real scored data.
