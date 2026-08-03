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

- **`model/prob_resid.py`** owns the model — the shape features (`SHAPE_FEATS`), how they are
  derived from raw kinematics (`add_shape_features`), the training routine
  (`train_global_model`), and scoring (`predict_global_rv`).
- **`model/train.py`** runs a full training pass and writes every inference artifact to
  `model/artifacts/`: the model (`ensemble_all.pkl`), movement baselines, spin-axis lookup,
  the cutter router, and normalization constants.
- **`model/predict.py`** (`StuffPlusPredictor`) loads those artifacts once and exposes
  `predict(df)`, which adds two columns: `stuff_plus` (the raw z-score used for aggregation)
  and `stuff_plus_display` (a percentile-anchored soft cap for individual pitches).

Because the model reads only its own artifacts and writes only those two columns, the model
can be replaced without touching ingestion, aggregation, or the web layer — which is exactly
what the v500 change below did.

## v500 — model responsibility change (Sprint 4)

The previous model (`v402_family_prob`) was **three separate family models** (fastball /
offspeed / breaking), each normalized on its own scale, composing
`P(whiff)·V_whiff + P(in-play)·P(HR|in-play)·V_HR` on induced-Magnus features.

It was replaced by **one global swing-outcome model** (see
[global-swing-model.md](global-swing-model.md)): a single swing softmax
`{whiff, foul, in-play}` plus a 5-cell contact grid, on raw-acceleration shape features, with
one global normalization scale.

The swap stayed inside the model boundary:

- `predict.py` now loads one artifact (`ensemble_all.pkl`) and applies one scale, instead of
  routing each pitch to one of three family models and three scales.
- `train.py` trains and saves one model instead of three.
- The cutter router and `assign_family` helpers are retained for pitch-family tagging but no
  longer gate scoring.
- Ingestion, aggregation, persistence, and the web app were unchanged apart from one comment
  and the feature-column list `app.py` derives from `SHAPE_FEATS` (which updates itself).

## Data and artifacts

- `data/statcast.db` — raw `pitches_<season>` and scored `pitches_<season>_scored` tables.
  Git-ignored; a mounted volume on Railway.
- `model/artifacts/ensemble_all.pkl` — the trained model (swing softmax + contact grid +
  run values + normalization). Committed.
- `model/artifacts/{movement_baselines,cutter_router,spin_axis_lookup,norm_global*}.pkl` —
  supporting inference artifacts. Committed.
- `model/artifacts/feature_cache.parquet` — the engineered training frame, rebuilt by
  `main.py train`. Git-ignored (large, regenerable).
- `profiles/output/**/*.json` — leaderboards and pitcher cards. Committed (PNGs are not).
