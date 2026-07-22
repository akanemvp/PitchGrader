# Method A — Two-Zone Conditional-Outcome Stuff+

## What it is
The production Stuff+ model (`model/prob_resid.py`). A pitch's expected run value is
split by **where its shape tends to go**, and each zone is scored by its own full
swing/take outcome tree:

```
xRV(shape) = P(in-zone | shape) · IN-ZONE value
           + P(out-zone | shape) · OUT-ZONE value
```

- **IN-ZONE value** = `P(swing | in-zone) · [whiff / foul / GB·AIR·PU·HR contact]`
  `+ P(take | in-zone) · V_called_strike` — an in-zone take is a **called strike**.
- **OUT-ZONE value** = `P(chase | out-zone) · [whiff / foul / GB·AIR·PU·HR contact]`
  `+ P(no-chase | out-zone) · V_ball` — an out-of-zone take is a **ball**.

Seven LightGBM heads, all on the 8 arm-normalized `SHAPE_FEATS`:
`zone` (P in-zone), `izgate` (P swing | in-zone), `iz_swing`, `iz_ip`, `chase`
(P swing | out-zone), `oz_swing`, `oz_ip`. Contact is a 4-cell trajectory model —
ground ball (LA<10), air/line-drive (10–50), pop-up (>50), home run — valued per cell
by count-adjusted run value, with chase contact valued weaker than in-zone contact.

## The key ideas (why this and not a flat model)
1. **Takes are credited to shape, fairly.** A called strike's value attaches through
   `P(take | in-zone, shape)` — how often a shape *freezes* hitters when it's in the
   zone (riding four-seams do this: `corr(in-zone take rate, IVB) = +0.26`). A ball's
   penalty attaches through `P(no-chase | out-zone, shape)`. Both heads are trained
   **only on pitches in their own zone**, so the credit isn't just "it was located
   there."
2. **The zone weight is command-neutral but accurate.** `P(in-zone | shape)` is trained
   on shape across all pitchers, so it's the *shape's* population-average zone tendency
   (fastball ~0.51, slider ~0.42, changeup ~0.35), not any individual's command. It's
   validated calibrated (AUC 0.58, ECE 0.01, per-type corr 0.97). A fixed zone rate was
   tried and rejected — it validated worse and inflated chase pitches.
3. **Location is a training-time label only.** In-zone/out-zone is defined against each
   hitter's individual strike zone (mean sz_top/sz_bot). At inference the seven heads
   are applied to shape alone — no location, count, or handedness as features.
4. **Chase contact is weaker.** A ball put in play on a chase is less damaging than one
   squared up in the zone; the two zones carry separate contact value tables.

## Why it's the production choice
Best-validated configuration built: it holds up against actual (held-out 2026) run
value (spearman ≈ −0.20), orders pitch types correctly (cross-type ≈ −0.3), keeps
fastballs *and* sliders un-compressed, and gives shape credit to called strikes
(in-zone) and balls (out-of-zone) the fair way. Its one structural blind spot is
changeups — their value is tunneling off the fastball, which a shape-only model can't
see (needs an arsenal-relative "vs. fastball" feature, a future lever).

## What did NOT change
- The 8 `SHAPE_FEATS`, the `stuff_plus` / `stuff_plus_display` scale, the composite
  (grade-the-average-pitch) aggregation, and the usage-weighted overall.
- The inference contract: `predict_prob_resid_rv(df, ens)` + `ensemble_all.pkl` +
  `norm_global.pkl`. The rest of the pipeline (train / score / profiles / live) is
  unchanged.

## Related change in this sprint
Arm-angle backfill (`live/live_update.py`): the live updater now re-scrapes a trailing
window every cycle and re-scores it, so the **real per-pitch Statcast arm_angle
replaces the live estimate** once Savant publishes it (a day or two post-game), instead
of the estimate being frozen in.
