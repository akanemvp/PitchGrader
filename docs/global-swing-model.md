# Stuff+ — global swing-outcome model (v500_global_swing_grid)

Replaces the three-family probability model ([family-prob-model.md](family-prob-model.md),
`v402`) with **one global model** trained on every pitch, in the style of Pitch Profiler /
proStuff+: shape only, no location, count, or game state.

## Model

    xRV = P(whiff)·V_whiff + P(foul)·V_foul + P(in-play)·E[contact RV | in-play]

Lower xRV = a better pitch. One global z-score turns xRV into a grade: **100 = league-average
pitch, 10 = one standard deviation**.

### Two heads

1. **Swing softmax** — a LightGBM multiclass head over `{whiff, foul, in-play}` on the 8 shape
   features (1000 trees, 31 leaves, lr 0.01).
2. **5-cell contact grid** — a LightGBM multiclass head that predicts which of five in-play
   cells a batted ball lands in:

   | cell | launch angle | exit velo | run value* |
   |------|-------------|-----------|-----------|
   | GB<95  | < 10° | < 95 | negative (good) |
   | GB95+  | < 10° | ≥ 95 | ~0 |
   | air<95 | 10–50° | < 95 | slightly positive |
   | air95+ | 10–50° | ≥ 95 | strongly positive (damage) |
   | pop    | ≥ 50° | — | negative (good) |

   *count-adjusted `delta_run_exp`, averaged empirically per cell. The grid is heavily
   smoothed (3000 min-child, strong regularization) because the per-pitch shape→cell map is
   noisy even though the per-cell values are steady; `E[contact RV | in-play]` is the grid's
   probability-weighted expectation.

### Valued outcomes

- `V_whiff ≈ -0.113`, `V_foul ≈ -0.039` — count-adjusted `delta_run_exp`, averaged over all
  whiffs / fouls. Both help the pitcher; a whiff helps more than a foul.
- In-play value is **not** a single number — it is the contact grid's expectation, so a pitch
  that yields weak grounders grades better in-play than one that yields barrels.

### Features (8, arm-normalized)

`release_speed, release_spin_rate, ax_arm, az, arm_angle, release_pos_x_arm, release_pos_z,
release_extension`

`ax_arm` (raw horizontal acceleration, arm-side signed), `az` (raw vertical acceleration), and
`release_pos_x_arm` (arm-side release point) are derived by `prob_resid.add_shape_features`
from raw kinematics and `p_throws`. Arm-signing means a lefty and a righty throwing physically
identical pitches receive identical features — one scoring pass serves both hands.

**Raw accelerations, not induced movement.** The prior model used induced-Magnus components
(`ind_vert` / `ind_horiz_arm`). Raw `az` / `ax_arm` are more predictive of actual whiff and
rank power arms above soft-tossers more faithfully, so the scoring model uses them; the induced
components are still computed for the cutter router.

## Scale

Grades share **one global scale**, so pitch types are directly comparable: breaking balls
(more whiffs, softer contact) average above fastballs, and fastballs (`FF`/`SI`) center below
100. This is the deliberate trade of a single-scale model — cross-type comparability at the
cost of per-type centering. (The retained `assign_family` helper still tags each pitch's
family for callers that want per-family views.)

## Files

- `model/prob_resid.py` — features, run values, swing softmax + contact grid, scoring.
- `model/train.py` — single global training pass, global (+ historical) normalization.
- `model/predict.py` — loads `ensemble_all.pkl`, scores all pitches on one scale.

## Artifacts

`ensemble_all.pkl` (swing softmax + grid + `Vcell` + `V_whiff`/`V_foul` + `norm`/`norm_hist`),
plus `cutter_router.pkl`, `movement_baselines.pkl`, `spin_axis_lookup.pkl`, `norm_global*.pkl`.
The old `ensemble_{fb,os,br}.pkl` are removed by training. Trained on 2022–2025.
