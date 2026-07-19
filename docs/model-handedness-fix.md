# Model Update — Handedness Normalization and Target Revision

*Sprint 3 engineering note. This documents a model-quality investigation and the
production change that came out of it.*

## The problem
Two left-handed relievers (Erik Miller, Adrián Morejón) graded 129.5 and 127.1 —
implausibly high, and higher than right-handers with objectively better inputs.
Edgardo Henriquez threw **harder** (100.8 vs 99.5 mph) with **more** arm-side force
(ax 20.6 vs 20.4) and graded **113.0**, fourteen points lower.

## Diagnosis: a mirror test
The model's shape features included the **raw, handedness-signed** `ax` (horizontal
acceleration) and `release_pos_x` (release side). For a right-hander, arm-side run is
*negative* `ax`; for a left-hander it is *positive*. So the model had to learn the
same physics **twice**, once per sign region — from data that is **73% right-handed**.

The test: take a real pitch, flip only the signs of `ax` and `release_pos_x` — no
physical change, just the mirror image — and re-grade it.

| | Mirror asymmetry |
|---|---|
| Old model (raw features) | **+4.72** Stuff+ (sinkers +6.25) |
| Fixed model (arm-normalized) | **0.00** |

A physically identical pitch graded ~5 points better thrown left-handed. Erik Miller
graded **129.5 as a lefty, 105.8 mirrored to a righty** — a 23.7-point swing from a
sign flip.

**Was the bonus real?** No. In the training outcomes, left-handers are slightly
*worse* overall (+0.00077 runs/pitch) and effectively tied on sinkers (≈0.3 Stuff+
points). The model was not learning that lefties perform better — it was fitting
sparse data. At `|ax| ≥ 20` there are **2.6× more right-handed examples**, and the
left-handed-vs-left-handed cell is only **7.5%** of all pitches (5.1× less than the
right-on-right cell).

**More data does not fix it.** Retraining on 2020–2025 (3.7M pitches, six seasons)
reproduced the bias in full. Adding seasons adds both hands in the same 73/27 ratio,
so the relative sparsity never improves. This is a representation problem, not a
sample-size problem.

## The fix
Mirror the two handedness-signed features so arm-side is positive for both hands:

```python
hand_sign          = -1 if p_throws == "R" else +1
ax_arm             = ax            * hand_sign
release_pos_x_arm  = release_pos_x * hand_sign
```

One physical pattern is now learned from 100% of the data, and handedness symmetry is
guaranteed **by construction** — not tuned. This matches published practice: tjStuff+
v3.0 mirrors the same two features for the same stated reason.

## Two further changes shipped alongside
**Spray-aware in-play target.** The old target converted xwOBAcon to runs, but
xwOBAcon knows only exit velocity and launch angle — it is blind to spray direction.
Measured consequence: sinker contact beat its xwOBAcon by ~23 wOBA points while
sweeper contact trailed by ~41, because grounders are hit toward fielders and pulled
air balls are not. Replacing it with a model on **(exit velo, launch angle, spray
angle)** improved out-of-sample prediction of in-play run value (**r 0.641 → 0.701**,
RMSE −7%) and cut pitch-type bias ~18%. It won all six head-to-head comparisons
against xwOBAcon, though the arsenal-level effect is under a point.

**Swing-only training.** Balls and called strikes are excluded (1.50M of 2.87M
pitches). Whether a taken pitch is a ball or a strike is mostly location and command,
which this model cannot see — it has no location features. *This is the least
validated of the three changes:* it materially reshuffles within-arsenal ordering
(fastballs down, breaking balls up) and we do not yet have evidence that the new
ordering is more correct. It should be revisited with a next-season predictive test.

**`same_hand` removed.** With mirrored features the model cannot identify pitcher
handedness, so platoon marginalization became a no-op (**+0.08** points league-wide).
Grades are now a single scoring pass instead of a same-hand/opposite-hand average.

## Training data
`TRAINING_SEASONS` is now **2022–2025**. Because 2025 is training data, it is no
longer scored or displayed — it would be an in-sample grade. 2026 and the
minor-league/college/spring seasons remain out-of-sample and are unaffected.

Training-window sensitivity was tested (2020–25, 2022–25, 2023–25): grades move ~1
point across a six-year span and the ranking never changes, so the window is not a
meaningful lever. Longer windows were mildly *worse* out-of-sample (r +0.1130 for
2020–25 vs +0.1213 for 2023–25), consistent with concept drift from the 2020–21 ball
and pre-sweeper-boom classification.

## Effect on grades
Left-handers come down, right-handers hold or rise — across seven arsenals tested,
with no exceptions:

| Pitcher | T | Old | New |
|---|---|---|---|
| Adrián Morejón | L | 122.5 | 112.8 |
| Dylan Cease | R | 106.1 | 107.7 |
| Clay Holmes | R | 108.3 | 107.4 |
| Erik Miller | L | 118.7 | 105.1 |
| J.T. Ginn | R | 100.5 | 104.2 |
| Tarik Skubal | L | 106.8 | 102.2 |
| Max Fried | L | 104.6 | 100.6 |

Erik Miller's sinker fell from 129.5 to 103.7 — roughly league average, which is what
his shape supports once handedness is handled correctly.

## Verification
- Mirror asymmetry: **+4.72 → 0.000** on the production model.
- Handedness gap on 2026: left-handers 99.2, right-handers 102.2 (the residual favors
  right-handers, so no left-handed premium remains).
- Scale intact: mean 101.3, SD 9.8 on 2026.
- Head-to-head predictive test (pitcher × pitch-type vs actual 2026 run value):
  arm-normalized **r = +0.1130** vs raw **r = +0.1090** on identical data — the fix
  improves accuracy as well as removing bias.

## Known limitations
- **Swing-only is unvalidated** (see above).
- **The fastball half of the xwOBA bias survives.** Spray fixed breaking-ball contact
  but fastball contact still outperforms the model's expectation, likely due to
  batted-ball spin, which Statcast does not expose.
- Grades remain **shape-only**: no command, sequencing, or location. A pitcher can
  substantially out-pitch his Stuff+ (Max Fried at ~100 is the clearest example here).
