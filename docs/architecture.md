# Architecture

*Sprint 2 — updated to reflect the working prototype. The original Sprint 1
architecture write-up remains at [docs/project/architecture.md](project/architecture.md).*

## Overview
PitchGrader is a **data pipeline plus a web app**. Raw pitch data is ingested and
stored, transformed into shape features, scored by a machine-learning model, and
summarized into per-pitcher profiles and leaderboards. A Flask web application serves
those results and lets a user re-label pitches and re-score them live.

## Major Components

**Client / User Interface** (`templates/`, `static/`)
- HTML + vanilla-JS pages that render the homepage (season dropdown + pitcher
  search), pitcher profile cards, leaderboards, the movement-profile plot, and the
  interactive pitch-type editor.

**Server / Application Logic**
- `app.py` — the Flask app: page routes, the JSON API the pages call, and the
  re-classification/re-score editor endpoints.
- `model/` — the scoring layer. Loads the trained model and turns pitch shape
  features into Stuff+ grades.
- `features/` — the feature-engineering layer. Computes pitch shape features from
  raw tracking measurements.
- `profiles/` — the profile/leaderboard builder. Aggregates scored pitches into
  per-pitcher, per-pitch-type cards and season leaderboards.
- `scraper/` — data ingestion from public pitch-tracking sources.
- `main.py` — a command-line entry point that runs training, scoring, and profile
  generation.

**Data / Persistence** (`data/`, `model/artifacts/`, `profiles/output/`)
- A SQLite database holding raw pitch tables (`pitches_<season>`), scored pitch
  tables (`pitches_<season>_scored`), and editor tables.
- Saved model artifacts: the trained model and the frozen league normalization.
- Generated JSON profile/leaderboard files the web app reads at request time.

## The Model
The production model is a **single LightGBM regressor** (`model/prob_resid.py`) that
predicts each pitch's expected run value from **8 shape features** — release speed,
extension, vertical acceleration, arm-normalized horizontal acceleration,
arm-normalized release side, release height, arm angle, and spin rate. It uses **no
location, no count, and no batter handedness**.

The two horizontal features are **mirrored** so arm-side is positive for both hands.
This matters: with raw signed values the model had to learn the same physics twice
(once per sign region) from data that is 73% right-handed, which gave left-handers a
~5-point phantom bonus. Mirroring makes a lefty and a righty throwing physically
identical pitches grade identically, by construction — see
[model-handedness-fix.md](model-handedness-fix.md).

The training target uses **swings only**: actual `delta_run_exp` for whiffs and fouls,
and a **spray-aware expected run value** (exit velo, launch angle, spray angle) for
balls in play. Balls and called strikes are excluded, since whether a taken pitch is
called a ball or a strike is mostly location and command — which this model cannot see.

Predicted run value is normalized onto the Stuff+ scale (**100 = league average,
10 = one standard deviation**) using frozen 2022–2025 Major League norms, so grades are
comparable across pitch types and seasons. Because 2025 is training data, it is no
longer scored or displayed.

**Aggregation:** every pitch is graded individually, but a pitch *type*'s score is the
grade of that type's **average pitch** (metrics averaged first, then graded) rather than
the average of the individual grades. The model is a tree ensemble, so
`mean(grade(x)) ≠ grade(mean(x))`, and averaging grades penalizes pitchers whose stuff
varies more from pitch to pitch.

## Data Flow
```
        Public pitch-tracking data (Statcast CSV / MLB gamefeed)
                            |
                       [ scraper/ ]
                            v
                Raw pitch tables  pitches_<season>            (SQLite)
                            |
                   [ features/ engineering ]
                            v
                 [ model/ scoring ]  <-- trained model + frozen norms
                            v
              Scored pitch tables  pitches_<season>_scored    (SQLite)
                            |
                   [ profiles/ builder ]
                            v
             Profile + leaderboard JSON  profiles/output/...
                            |
                        [ app.py ]
                            v
                    Browser (profiles, leaderboards, editor)
```

## Sprint 2 Feature Slice — the pitch-type editor & live re-score
The complete vertical slice demonstrated this sprint is **re-classify → re-score →
preview**:

1. On a pitcher's profile the user opens the **editor** and relabels one or more
   pitches (e.g. a mislabeled "slider" that is really a sweeper).
2. The page POSTs the overrides to `/api/reclassify_profile` (`app.py`).
3. The server applies the new labels, **re-runs the model** on those pitches
   (`StuffPlusPredictor.predict`), re-aggregates with `summarize_pitcher`, and
   returns updated grades **plus per-pitch movement dots carrying the new labels** —
   all **without writing to the database** (a safe preview).
4. The profile page re-renders with the new grades, and the movement-profile chart
   recolors its dots to the re-classified pitch types.

This slice exercises every layer — UI, web route, feature engineering, model
inference, and aggregation — which is why it is the right size for Sprint 2.

## Seasons / Levels
The same pipeline ingests and scores multiple levels by season key: MLB (`2026`),
Triple-A (`aaa2026`), the ACL and Florida State League, spring training, and — added
this sprint — **tracked NCAA college games** (`college2026`, MLB StatsAPI sportId 22).
Completed seasons are scored once with historical norms; live seasons refresh
periodically.
