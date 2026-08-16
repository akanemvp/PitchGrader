# AI-assisted implementation review

The Sprint 5 model (`v700_threegroup_gbair`) — the three-group model, the removal of the
exit-velocity contact head, the rewritten tests, and this documentation — was produced with an
AI pair (Claude Code). This records how AI was used for exploration, measurement, test
generation, and cleanup, what was revised or rejected, and how the AI-generated work was
verified before it was trusted.

## AI-assisted model exploration — the contact head

The central Sprint 5 question was whether the model's exit-velocity contact machinery earned
its complexity. Rather than argue it, the AI was directed to **measure it out-of-sample** (train
on ≤2025, test on 2026), and the evidence — not intuition — drove the decision:

- **Does shape predict exit velocity?** Barely — Pearson r = 0.18, r² = 0.03. Predicted EV
  spanned ~2.4 mph against 15.3 mph of real spread. Exit velocity is overwhelmingly
  batter-controlled; it is not a repeatable pitch-shape signal.
- **Did the EV term add ranking skill?** No. With ground-ball probability held fixed, the
  EV-only contact score correlated −0.05 with actual contact run value — statistical noise. The
  full contact score turned out to be **93% just P(GB)**.
- **Would a plain GB/air classifier do as well?** Yes — P(GB) alone ranked contact damage as
  well or better than the full EV apparatus (Spearman ≈ 0.19–0.23).

**Decision:** the exit-velocity heads (per-type EV-distribution regressors + empirical EV→RV
curves — dozens of models) were **removed**, and the in-play term reduced to
`P(GB)·V_gb + P(air)·V_air`. The value here was the AI running the measurement quickly and
following the numbers to a simpler, better-justified model.

A related finding the AI surfaced and the team accepted: because pitch shape *cannot* recover
exit velocity, an "exit velocity over expected" idea (a residual metric popular in public
analysis) would belong in a separate *pitcher*-level layer, not inside this shape-only Stuff+.

## AI-assisted architecture experiments

The AI built and boarded several structural variants side by side so the shipped design was
chosen against real comparisons, not a hunch:

- **Single model vs 2-group (FB/NONFB) vs 3-group (FB/BR/OFF) split**, all on one shared scale.
  The split's measurable effect (once EV was gone) was concentrated in a few pitches — most
  visibly cutters, which route to a dedicated group and grade more sharply there. The 3-group
  split was kept for its cleaner within-family resolution.
- **Per-group scales and per-group run values — rejected.** A whiff should be worth the same run
  value regardless of the pitch that produced it; the shared scale + global values keep grades
  comparable across pitch types, which the product wants.
- **A pop-up (GB/air/pop-up) contact head and an HR-in-air head** were prototyped and boarded;
  they re-tilted the fastball/sinker footing without a clear predictive win, and were not
  shipped. (Pop-up propensity *is* shape-predictable, unlike exit velocity — noted for future
  work.)

## A correctness bug the AI found while documenting

The earlier model required Statcast's `spin_axis` (its features used a Magnus/non-Magnus split
derived from the measured spin axis). The current model's induced features come from the
pitch's **kinematics** instead. While writing the docs the AI asserted the old "needs
spin_axis" property, then **verified it empirically** rather than trusting the prior text — and
found the same pitch grades to **94.6 with `spin_axis` and 94.6 without it**. The dependency was
gone. The docs, the module docstrings, and two stale tests
(`test_spin_axis_required_to_score`) were corrected to state the true behavior: the model scores
from kinematics alone, so minor-league feeds without a spin axis now score. This is a case of
**verifying a claim before documenting it**, and the verification changed the claim.

## AI-assisted test generation

The test suite was rewritten this sprint to match the new model core:

- **Pure-function tests** (`test_prob_resid.py`) — the induced-feature derivation, arm-side and
  induced-horizontal normalization (a mirror-image lefty and righty grade identically), the
  binary GB/air labelling, outcome run-value signs (whiff < foul < 0, GB < 0 < air), and
  FB/BR/OFF routing. No trained model needed, so they run in ~1 s.
- **Behavioral tests** (`test_predict.py`) — kinematics-only scoring (with and without
  `spin_axis`), velocity monotonicity, all three groups scoring, and bounded finite grades.
  These load `ensemble_all.pkl` and `skipif` it is absent, so a fresh checkout still passes.

The AI was directed to test *properties* (monotone, bounded, arm-symmetric, NaN-when-unscorable)
rather than frozen numeric outputs, so the suite survives a retrain without golden-value churn.
One first-draft test asserted an incomplete handedness mirror (it reflected `ax` but not `vx0`
or the release point) and failed on the real data; it was fixed to a full x-axis reflection.

## Dead-code and documentation cleanup

With the EV heads and the Magnus/non-Magnus split gone, the AI removed the now-unreachable code
it left behind — the EV→RV curve helper, the quantile list, the pop-up cell names, and unused
sampling constants — and rewrote the module docstrings in `prob_resid.py`, `train.py`, and
`predict.py`, which still described the old per-group-scale SIERA model. All six required docs
were brought in line with the shipped model.

## How AI-generated work was verified

- **Ran the suite:** `python -m pytest tests/` → **19 passed**.
- **End-to-end retrain:** `python main.py train` completed with the expected global values and
  three-group logs, and saved every artifact.
- **Board sanity check:** the 2026 board was compared against the eye test (Misiorowski / Miller
  lead the four-seams, Alvarado's cutter routes to the breaking group and tops the cutters, no
  outliers) before accepting.
- **Empirical verification of claims:** the `spin_axis` independence and the EV-vs-GB contact
  finding were both confirmed by direct measurement, not assumed from prior text.

No AI-generated code, test, or documentation claim was committed until it had passed the suite
and been checked against real scored data.
