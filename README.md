# PitchGrader — MLB Stuff+

PitchGrader grades the raw nastiness ("Stuff+") of every MLB and minor-league pitch from
its physical shape — velocity, spin, movement, release, and arm slot — independent of where
it was located or the count it was thrown in. A grade of **100 is league average** and
**every 10 points is one standard deviation**, so a 120 fastball is a top-decile fastball.

## What it does

- **Scores every pitch** in 2026 MLB plus AAA, spring, breakout, ACL, FSL, college, and
  Futures feeds.
- **Builds pitcher cards and leaderboards** — per-pitch-type Stuff+, arsenal breakdowns, and
  season lines pulled from official boxscores.
- **Updates live** during the season (a background worker re-scrapes and re-scores in-progress
  games).
- **Lets you fix pitch-type labels** in a web editor; corrections persist and are re-applied
  on every re-score.

## The model

Stuff+ is trained on ~2.9 M pitches (2022–2025) and scores 2026 fully out-of-sample. It turns
pitch shape into an expected run value, then into a grade. It is a Pitch Profiler / proStuff+-
style model: shape only, no location or count.

    xRV = P(whiff)·V_whiff + P(foul)·V_foul + P(in-play)·[ P(GB)·V_gb + P(air)·V_air ]

- **One unified model** over all pitch types — fastballs, breaking balls, and offspeed are
  graded by a single set of heads on one shared scale, so a fastball, breaking ball, and
  offspeed pitch are directly comparable. Breaking balls (more whiffs, softer contact) sit
  above fastballs on average.
- A **swing softmax** head predicts `{whiff, foul, in-play}` from shape.
- A **GB/air contact head** predicts whether an in-play ball is a ground ball (`<10°`) or in
  the air; the in-play term weights those by two run-value constants. (Exit-velocity heads were
  tried and removed — pitch shape explains ~3% of exit-velocity variance and added no
  out-of-sample skill; the repeatable contact signal is ground-ball propensity.)
- **8 arm-normalized shape features**: velocity, spin rate, induced (Magnus-frame) vertical and
  arm-side horizontal acceleration, arm angle, release side and height, and extension. Features
  are arm-signed, so a lefty and a righty throwing physically identical pitches grade
  identically. Movement is derived from the pitch's kinematics, so a pitch scores without
  needing Statcast's `spin_axis`. Extension and arm angle are winsorized to physically
  plausible bands to keep tracking errors from distorting the fit.
- Heads are shallow **LightGBM** regressors/classifiers with `linear_tree` splits. Grades are a
  single shared z-score of the expected run value, so every pitch is measured against every
  other pitch.

## Running it

```bash
python -m pip install -r requirements.txt

python main.py train      # train the model from the season tables -> model/artifacts/
python main.py score      # score each season's raw pitches -> pitches_<season>_scored
python main.py profiles   # aggregate scored pitches into cards + leaderboards (JSON)
python main.py live       # start the live in-season update loop

./start.sh                # run the Flask web app locally on http://localhost:5001
gunicorn app:app          # serve the web app (see Procfile for the Railway command)
```

Training data lives in a SQLite database at `data/statcast.db` (git-ignored; on Railway it is
a mounted volume). The trained model is `model/artifacts/ensemble_all.pkl` plus the movement
baselines, cutter router, spin-axis lookup, and normalization files alongside it.

## Layout

| Path | Responsibility |
|------|----------------|
| `model/prob_resid.py` | the model: features, swing softmax, GB/air contact head, run values |
| `model/train.py` | training run — fits and saves all inference artifacts |
| `model/predict.py` | inference — loads the model, turns pitches into grades |
| `features/engineering.py` | shape-feature engineering + induced (Magnus) components |
| `main.py` | CLI: `train` / `score` / `profiles` / `live` |
| `app.py` | Flask web app (leaderboards, pitcher cards, editor) |
| `profiles/` | pitcher-card and leaderboard builders |
| `scraper/`, `live/` | data ingestion and the live update loop |
