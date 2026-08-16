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
| Feature engineering | `features/engineering.py` | Turn raw kinematics into the model's shape features (movement baselines, arm angle, induced-Magnus components). |
| Model | `model/prob_resid.py`, `model/train.py`, `model/predict.py` | Train the Stuff+ model and score pitch shape into a grade. |
| Aggregation | `profiles/` | Roll scored pitches up into per-pitcher, per-pitch-type cards and season leaderboards. |
| Persistence | `data/statcast.db`, `storage/overrides.py` | SQLite tables; durable pitch-type override table re-applied on every re-score. |
| Web | `app.py`, `templates/` | Leaderboards, pitcher cards, and the pitch-type editor. |
| CLI | `main.py` | `train` / `score` / `profiles` / `live`. |

## The model boundary

Everything above the model treats a pitch as a row of raw Statcast columns; everything the
model needs is derived inside `model/`. The contract is small and stable:

- **`model/prob_resid.py`** owns the model — the 8 arm-normalized shape features
  (`SHAPE_FEATS`), how they are derived from raw kinematics (`add_magnus` for the induced
  Magnus-frame movement, `add_shape_features` for the arm-signed columns), the three-group
  training routine (`train_split_model`), and scoring (`predict_group_rv` → raw xRV,
  `grade_pitches` → the shared z-score).
- **`model/train.py`** runs a full training pass and writes every inference artifact to
  `model/artifacts/`: the model (`ensemble_all.pkl`), movement baselines, spin-axis lookup,
  the cutter router, and normalization constants.
- **`model/predict.py`** (`StuffPlusPredictor`) loads those artifacts once and exposes
  `predict(df)`, which adds two columns: `stuff_plus` (the raw z-score used for aggregation)
  and `stuff_plus_display` (a percentile-anchored soft cap for individual pitches).

Because the model reads only its own artifacts and writes only those two columns, the model
can be replaced without touching ingestion, aggregation, or the web layer.

## v700 — three-group probability model

The model splits pitches into **three groups** — fastballs (FB), breaking balls (BR), and
offspeed (OFF) — with cutters (FC) routed to FB or BR by a Mahalanobis classifier. Each group
trains its own heads on its own subset, which lets a group learn a shape→outcome mapping
specific to its family; but all three share **one grading scale** and **one global set of run
values**, so a fastball, breaking ball, and offspeed pitch are graded on the same scale.

Each group is:

- a **swing softmax** head — `{whiff, foul, in-play}` from shape; and
- a **GB/air contact head** — a binary classifier for whether an in-play ball is a ground ball
  (`<10°`) or in the air (any non-ground-ball contact).

The expected run value of a pitch is

    xRV = P(whiff)·V_whiff + P(foul)·V_foul + P(in-play)·[ P(GB)·V_gb + P(air)·V_air ]

where the four `V_*` are **global constants** (mean `delta_run_exp` per outcome over all
pitches). Per-pitch differentiation comes from the probabilities, not the values — a whiff is
worth the same run value no matter which pitch produced it.

**Shape** is 8 arm-normalized features: velocity, spin rate, the induced (Magnus-frame,
gravity + drag removed) vertical (`ind_vert`) and arm-side horizontal (`ind_horiz_arm`)
accelerations, arm angle, release side (`release_pos_x_arm`), release height, and extension.
The induced movement is computed from the pitch's 9-parameter trajectory (velocity and
acceleration vectors), so — unlike the earlier Magnus/non-Magnus split — **a pitch scores from
kinematics alone and does not need Statcast's `spin_axis`** (minor-league feeds without it
still score). Features are arm-signed, so a mirror-image lefty and righty grade identically.

The heads are **LightGBM** (`num_leaves=8`, `max_depth=3`, `linear_tree`), each **early-stopped**
on a held-out split (round count picked on 15% holdout, then refit on all data); features are
passed through unscaled.

### What was tried and removed (see docs/ai-implementation-review.md)

- **Exit-velocity heads** (per-type EV-distribution regressors → run-value curves): removed.
  Pitch shape explains ~3% of exit-velocity variance and the EV term added no out-of-sample
  contact-ranking skill; a plain GB/air classifier ranked contact damage as well or better.
- **Per-group scales and per-group run values**: rejected in favor of one shared scale + global
  values — a whiff/grounder/fly is worth the same regardless of shape.

### How the swap stayed inside the model boundary

- `predict.py` routes each pitch to its group model and normalizes on the one shared scale
  (`grade_pitches`).
- `train.py` trains three group models and saves the supporting inference artifacts.
- Ingestion, aggregation, persistence, and the web app are unchanged.

## Data and artifacts

- `data/statcast.db` — raw `pitches_<season>` and scored `pitches_<season>_scored` tables.
  Git-ignored; a mounted volume on Railway.
- `model/artifacts/ensemble_all.pkl` — the trained three-group model (three group models, each
  a swing softmax + GB/air contact head, the global run values, and the shared normalization).
  Committed.
- `model/artifacts/{movement_baselines,cutter_router,spin_axis_lookup,spin_offset}.pkl` —
  supporting inference artifacts. Committed.
- `model/artifacts/feature_cache.parquet` — the engineered training frame, rebuilt by
  `main.py train`. Git-ignored (large, regenerable).
- `profiles/output/**/*.json` — leaderboards and pitcher cards. Committed (PNGs are not).
