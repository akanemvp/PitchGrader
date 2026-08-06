# Architecture

PitchGrader is a data pipeline with a web front end. Raw pitches are scraped into a SQLite
database, a machine-learning model scores each pitch's shape, the scored pitches are
aggregated into pitcher cards and leaderboards, and a Flask app serves them.

```
scrapers ─▶ data/statcast.db ─▶ main.py score ─▶ *_scored tables ─▶ main.py profiles ─▶ JSON cards
   (raw pitches_<season>)          (model)         (per-pitch grades)      (aggregation)     + leaderboards
                                       ▲                                                          │
                                       │                                                          ▼
                          model/artifacts/ensemble_all.pkl                                    app.py (Flask)
```

## Responsibilities

| Layer | Module(s) | Responsibility |
|-------|-----------|----------------|
| Ingestion | `scraper/`, `live/` | Scrape Statcast / Savant / MLB feeds into `pitches_<season>` tables; the live worker re-scrapes in-progress games. |
| Feature engineering | `features/engineering.py` | Turn raw kinematics into the model's shape features (movement baselines, arm angle, Magnus components). |
| Model | `model/prob_resid.py`, `model/train.py`, `model/predict.py` | Train the Stuff+ model and score pitch shape into a grade. |
| Aggregation | `profiles/` | Roll scored pitches up into per-pitcher, per-pitch-type cards and season leaderboards. |
| Persistence | `data/statcast.db`, `storage/overrides.py` | SQLite tables; durable pitch-type override table re-applied on every re-score. |
| Web | `app.py`, `templates/` | Leaderboards, pitcher cards, and the pitch-type editor. |
| CLI | `main.py` | `train` / `score` / `profiles` / `live`. |

## The model boundary

Everything above the model treats a pitch as a row of raw Statcast columns; everything the
model needs is derived inside `model/`. The contract is small and stable:

- **`model/prob_resid.py`** owns the model — the 10 Magnus/non-Magnus shape features
  (`SHAPE_FEATS`), how they are derived from raw kinematics and the measured spin axis
  (`add_shape_features`), the two-group training routine (`train_split_model`), and scoring
  (`predict_group_rv` → raw xRV, `grade_pitches` → per-group z-score).
- **`model/train.py`** runs a full training pass and writes every inference artifact to
  `model/artifacts/`: the model (`ensemble_all.pkl`), movement baselines, spin-axis lookup,
  the cutter router, and normalization constants.
- **`model/predict.py`** (`StuffPlusPredictor`) loads those artifacts once and exposes
  `predict(df)`, which adds two columns: `stuff_plus` (the raw z-score used for aggregation)
  and `stuff_plus_display` (a percentile-anchored soft cap for individual pitches).

Because the model reads only its own artifacts and writes only those two columns, the model
can be replaced without touching ingestion, aggregation, or the web layer — which is exactly
what the split change below did.

## v600 — fastball / non-fastball split (Sprint 4)

The previous model (`v500_global_swing_grid`) was **one global model on a single scale**,
which buried fastballs under breaking balls: breaking balls miss more bats, so they averaged
above every four-seam and no fastball could stand out.

It was replaced by a **fastball / non-fastball split** — two models, cutters routed by the
Mahalanobis classifier — each **normalized on its OWN scale**, so a fastball is graded against
fastballs and a breaking ball against breaking balls. Each model is a swing softmax
`{whiff, foul, in-play}` plus a **SIERA-style contact head**: a GB/FB/PU classifier whose
in-play value is `a·(P(GB) − P(FB) − P(PU)) + b`, with **line drives excluded** from training
(line-drive rate is a batter/luck outcome, not a repeatable pitcher skill). Shape is now the
10 **Magnus/non-Magnus** features derived from the measured 3D spin axis (Nathan's method), so
a pitch is unscorable until Statcast fills its `spin_axis` in. Features are RobustScaler-
standardized before the linear-tree heads.

The swap stayed inside the model boundary:

- `predict.py` routes each pitch to its group model and normalizes on that group's scale
  (`grade_pitches`), instead of one global scale.
- `train.py` trains two group models and saves the per-hand spin-axis convention offset
  (`spin_offset.pkl`) that the Magnus split needs at inference.
- `app.py` adds `spin_axis` to the columns it pulls (the Magnus split depends on it);
  ingestion, aggregation, persistence, and the web app are otherwise unchanged.

## Data and artifacts

- `data/statcast.db` — raw `pitches_<season>` and scored `pitches_<season>_scored` tables.
  Git-ignored; a mounted volume on Railway.
- `model/artifacts/ensemble_all.pkl` — the trained split model (two group models: swing
  softmax + SIERA contact head, the RobustScaler, run values, and per-group normalization).
  Committed.
- `model/artifacts/{movement_baselines,cutter_router,spin_axis_lookup,spin_offset}.pkl` —
  supporting inference artifacts (`spin_offset` is the per-hand Magnus convention offset).
  Committed.
- `model/artifacts/feature_cache.parquet` — the engineered training frame, rebuilt by
  `main.py train`. Git-ignored (large, regenerable).
- `profiles/output/**/*.json` — leaderboards and pitcher cards. Committed (PNGs are not).
