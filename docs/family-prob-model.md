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

## v401 — HR-head regularization (per-pitch smoothing)

Home runs are rare (~4.5% of balls in play), so at light regularization the binary HR
head overfit feature *combinations*: smooth single-feature marginals but jumpy joint
predictions that cratered individual pitches near a pitcher's own average shape (e.g. a
near-mean Glasnow four-seam grading 75 while his composite was 107, driven by a spurious
P(HR)=0.09 vs the mean shape's 0.024).

Fix: the HR head now trains with heavy regularization (`min_child_samples=800`,
`num_leaves=15`, `max_depth=6`, `path_smooth=1.0`, 150 trees). P(HR) now varies gently
with shape — per-pitch spread on a fixed pitcher drops ~30% (Glasnow FF std 11.9 → 8.2,
min 55 → 87) with the composite/leaderboard unchanged. The whiff/multiclass head was
already smooth and is untouched. Foul remains an (unvalued) class.

## v402 — raw (un-normalized) horizontal features

`release_pos_x` and horizontal Magnus break are now used RAW (`ind_horiz`) instead of
arm-normalized (`release_pos_x_arm` / `ind_horiz_arm`), and `spin_eff` was dropped after
it landed at ~8% importance and didn't separate similar-shape fastballs.

Why: arm-normalization mirrors lefties onto righties so a mirror pitch grades identically —
but that pooled away real L/R signal. Un-normalized, the model reads release side and break
direction directly and separates fastballs the mirrored view could not (e.g. Mason Montgomery
FF, a lefty, now correctly out-grades Andrew Painter's — P(whiff) 0.128 vs 0.100, matching
their actual 0.259 vs 0.078 whiff/swing; the arm-normalized model had both stuck at ~0.103).

Trade-off: the model is now handedness-dependent — a lefty and righty throwing mirror-image
pitches grade slightly differently. The cutter router still uses the arm-signed `ind_horiz_arm`
(handedness-robust routing); `_composite_pitch_grade` carries the router features through.
