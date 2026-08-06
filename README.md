# PitchGrader — MLB Stuff+

PitchGrader grades the raw nastiness ("Stuff+") of every MLB and minor-league pitch from
its physical shape — velocity, spin, movement, release, and arm slot — independent of where
it was located or the count it was thrown in. A grade of **100 is league average** and
**every 10 points is one standard deviation**, so a 120 fastball is a top-decile fastball.


## What it does

- **Scores every pitch** in 2026 MLB plus AAA, spring, ACL, FSL, college, and Futures feeds.
- **Builds pitcher cards and leaderboards** — per-pitch-type Stuff+, arsenal breakdowns, and
  season lines pulled from official boxscores.
- **Updates live** during the season (a background worker re-scrapes and re-scores in-progress
  games).

## The model (v500 — global swing-outcome model)

Stuff+ is one model, trained on ~2.9 M pitches (2022–2025), that turns pitch shape into an
expected run value and then into a grade. It is a Pitch Profiler / proStuff+-style model:
shape only, no location or count.

    xRV = P(whiff)·V_whiff + P(foul)·V_foul + P(in-play)·E[contact run value | in-play]

- A **swing softmax** head predicts `{whiff, foul, in-play}` from shape.
- A **5-cell contact grid** (ground ball / air ball, each split at 95 mph exit velocity, plus
  pop-ups) predicts where an in-play ball lands; each cell carries its empirical run value, so
  the in-play term is the grid's expectation.
- **8 arm-normalized shape features**: velocity, spin rate, raw vertical and arm-side
  horizontal acceleration, arm angle, release side and height, and extension. Features are
  arm-signed, so a lefty and a righty throwing physically identical pitches grade identically.
- Grades are **one global z-score**, so a pitch is measured against every pitch — breaking
  balls (more whiffs, softer contact) sit above fastballs on average.

See [docs/architecture.md](docs/architecture.md) for the full pipeline and
[docs/global-swing-model.md](docs/global-swing-model.md) for the model rationale.

## Running it

```bash
python -m pip install -r requirements.txt

python main.py train      # train the global model from the season tables -> model/artifacts/
python main.py score      # score each season's raw pitches -> pitches_<season>_scored
python main.py profiles   # aggregate scored pitches into cards + leaderboards (JSON)
python main.py live       # start the live in-season update loop

gunicorn app:app          # serve the Flask web app (see Procfile for the Railway command)
```

Training data lives in a SQLite database at `data/statcast.db` (git-ignored; on Railway it is
a mounted volume). The trained model is `model/artifacts/ensemble_all.pkl` plus the movement
baselines, cutter router, and normalization files alongside it.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/
```

The suite covers the model core (feature arm-signing, contact-grid labelling, outcome run
values) and behavioral invariants of the trained model (velocity monotonicity, left/right
handedness symmetry, and the eye-test that a 100 mph riding four-seam out-grades a flat 91).
Tests that need the trained artifact skip automatically on a fresh checkout.

## Layout

| Path | Responsibility |
|------|----------------|
| `model/prob_resid.py` | the model: features, swing softmax, contact grid, run values |
| `model/train.py` | training run — fits and saves all inference artifacts |
| `model/predict.py` | inference — loads the model, turns pitches into grades |
| `features/engineering.py` | shape-feature engineering + Magnus components |
| `main.py` | CLI: `train` / `score` / `profiles` / `live` |
| `app.py` | Flask web app (leaderboards, pitcher cards, editor) |
| `profiles/` | pitcher-card and leaderboard builders |
| `scraper/`, `live/` | data ingestion and the live update loop |
| `tests/` | automated tests |
| `docs/` | architecture and model notes |
