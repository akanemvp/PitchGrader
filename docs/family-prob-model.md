# Stuff+ — 3-family probability model (v400_family_prob)

Replaces the single direct-RV regressor (`v300_direct_rv`) with **three family models**
— fastball, offspeed, breaking — each trained and normalized on its **own scale**.

## Why

The single model graded every pitch on one run-value scale, where a splitter legitimately
out-scores a four-seam per pitch — so fastballs looked systematically "shafted" as a class.
Grading each pitch **against its own family** removes that: a four-seam is scored against
four-seams, so FF/SI center at 100 like everything else, while the within-family ordering
that matches the eye test is preserved (power fastballs — Misiorowski, Miller, Alvarado —
lead the four-seams).

## Model

Each family is a **probability model** on the 8 arm-normalized induced-Magnus shape features:

    xRV = P(whiff)·V_whiff + P(in-play)·P(HR | in-play)·V_HR

- Multiclass head `{other, whiff, foul, in-play}` + binary `P(HR | in-play)` head (LightGBM).
- Foul is a class (proper probabilities) but **not valued** — only whiff and in-play home-run
  damage drive the grade, which keeps low-slot / high-ride power fastballs on top.
- Global count-adjusted values: `whiff ≈ -0.113`, `in-play HR ≈ +1.53`.
- Each family self-normalizes to 100 = family average, 10 = one SD (`ens["norm"]`,
  `ens["norm_hist"]` for historical seasons).

### Features (8)
`release_speed, release_extension, release_pos_x_arm, release_pos_z, arm_angle,
release_spin_rate, ind_vert, ind_horiz_arm`

`ind_vert` / `ind_horiz_arm` are the **induced-Magnus acceleration** components (spin-only
lift; `az+g` minus the drag-parallel part, arm-side signed), added by
`features.engineering.add_magnus` (also backfilled onto the training cache at train time).
This replaces the old raw `az` / `ax_arm` inputs.

### Cutter router
Statcast labels every cutter `FC`, but some behave like fastballs and some like sliders. A
Mahalanobis router (`model/artifacts/cutter_router.pkl`) compares each cutter's velocity +
arm-relative movement to the fastball-family and breaking-family centroids and routes it to
the nearer one, **per-pitcher majority vote**. Slider-cutters (e.g. Crochet) are graded on the
breaking scale, not the fastball scale.

## Files
- `model/prob_resid.py` — family probability model, values, cutter router, family assignment.
- `model/train.py` — 3-family training loop, router fit, per-family (+ historical) norms.
- `model/predict.py` — loads `ensemble_{fb,os,br}.pkl` + router, routes each pitch, scores on
  its family scale.
- `features/engineering.py` — `add_magnus` (ind_vert / ind_horiz_arm).

## Artifacts
`ensemble_fb.pkl`, `ensemble_os.pkl`, `ensemble_br.pkl`, `cutter_router.pkl`.
Old `ensemble_all.pkl` removed. Trained on 2022–2025; all live-season profiles regenerated.
