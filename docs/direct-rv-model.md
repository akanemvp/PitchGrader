# Stuff+ — Direct Run-Value Regressor

## What it is
The production Stuff+ model (`model/prob_resid.py`). A pitch's grade is its **expected
run value given shape alone**, produced by a single LightGBM regressor:

```
target(pitch) = mean count-adjusted RV of its outcome bucket
xRV(shape)    = regressor(shape)          # E[RV | shape]
stuff_plus    = 100 + (mean − xRV) / std · 10        (lower xRV = better)
```

Every pitch is assigned the **league-mean count-adjusted run value of its outcome
bucket**, and one regressor is fit to predict that value from the 8 arm-normalized
`SHAPE_FEATS`. The six buckets (each pitch falls in exactly one):

| bucket | mean RV (2022–25) |
|---|---|
| whiff | ≈ −0.11 |
| foul | ≈ −0.04 |
| called strike (in-zone take) | ≈ −0.054 |
| ball (out-of-zone take) | ≈ +0.050 |
| in-play, non-HR | ≈ −0.018 |
| in-play, home run | ≈ +1.53 |

Called-strike vs ball is the only place location enters — an in-zone take is a called
strike, an out-of-zone take is a ball — labeled at **train time** against the
individual hitter's strike zone. Whiff/foul/in-play values are global (not zone-split).
Contact is **home-run vs not** only; a finer GB/AIR/PU grid and a full 4-cell trajectory
model were tried and lost — shape can't predict exit velocity, so the extra cells add
noise. Trained on 2022–2025.

## Why this replaced the 7-head structured model
The previous production model composed seven conditional-probability heads
(`zone`, `izgate`, `iz_swing`, `iz_hr`, `chase`, `oz_swing`, `oz_hr`) into the same xRV.
Fitting a single regressor **directly** to `E[RV | shape]` is both simpler and
better-calibrated on the decisive test — cross-year predictiveness (year-N grade →
year-N+1 outcomes, 2025→2026, IP≥60, n=94):

| Outcome | 7-head model | **Direct** |
|---|---|---|
| FIP | −0.504 | **−0.547** |
| SIERA | −0.499 | **−0.520** |
| K-BB% | 0.453 | 0.473 |
| xERA | −0.470 | −0.514 |
| ERA | −0.340 | −0.387 |

Two findings drove the switch:
1. **The mean-outcome target beats raw run value.** Regressing on each pitch's own
   `delta_run_exp` (version B) collapsed FIP to −0.464 — per-pitch RV is dominated by
   sequence luck. Collapsing every outcome to its league-mean RV strips that noise.
2. **One joint regressor beats the cascade.** Fitting seven heads independently and
   multiplying introduces small joint-calibration errors that a single fit avoids. The
   decomposition bought interpretability (chase %, per-zone splits) at a slight
   predictive cost. We took the predictiveness.

The regressor uses light smoothing (`path_smooth=0.5`, `min_child_samples=400`) with a
fixed-seed 85/15 split and early stopping, converging around ~380 trees.

## Structural blind spots (unchanged from before)
- **Changeups / tunneled sliders** grade on raw shape only. Their value is separation
  off the fastball, which a pure-shape model can't see. An arsenal-relative "vs.
  fastball" feature (Δvelo / Δmovement) was tested and clearly lifts predictiveness
  (K-BB% 0.473 → 0.519, beats proStuff+ on FIP/xERA) but abandons the pure-shape
  principle — a changeup's grade would then depend on which fastball it's paired with.
  Held as a deliberate future lever, not shipped.
- **Spin efficiency** (Nathan trajectory method) was tested and is a wash — helps K-BB%
  slightly, costs FIP.

## The inference contract (unchanged)
`predict_prob_resid_rv(df, ens)` returns xRV; `ens["feats"]` = the 8 `SHAPE_FEATS`;
`ens["regressor"]` is the fitted model. `stuff_plus` / `stuff_plus_display`, the
composite (grade-the-average-pitch) aggregation, the usage-weighted overall, and the
train / score / profiles / live pipeline are all unchanged. `norm_global.pkl` and
`norm_global_historical.pkl` are regenerated at train time for the new xRV scale.
