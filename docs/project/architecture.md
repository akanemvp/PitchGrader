# Architecture

## Overview
The project is organized as a **data pipeline plus a web app**. Raw pitch data is
ingested and stored, transformed into shape features, scored by a machine-learning
model, and summarized into profiles and leaderboards. A Flask web application then
serves those results to the user and lets them re-label pitches and re-score.

## Major Components

**Client / User Interface**
- HTML templates and static JavaScript/CSS that render pitcher profile pages,
  leaderboards, movement plots, and the pitch-type editor.

**Server / Application Logic**
- A Flask app (`app.py`) that handles web routes and the re-score editor endpoint.
- A scoring layer (`model/`) that loads the trained model and turns pitch features
  into Stuff+ grades.
- A feature-engineering layer (`features/`) that computes pitch shape features.
- A profile/leaderboard builder (`profiles/`) that summarizes scored pitches.
- A command-line entry point (`main.py`) that runs scoring and profile generation.

**Data / Persistence**
- A SQLite database holding raw pitch tables, scored pitch tables, and editor tables.
- Saved model artifacts (the trained model and the frozen league norms).
- Generated JSON profile/leaderboard files used by the web app.

## Component Responsibilities
- **Data ingestion (scraper):** pull pitch data from public sources and load it into
  the raw pitch tables.
- **Feature engineering:** convert raw measurements into the shape features the model
  needs (velocity, movement, spin, release, arm angle).
- **Model / scoring:** assign each pitch a Stuff+ value and normalize it onto the
  100 / 10-per-SD scale using the frozen norms.
- **Profile builder:** aggregate scored pitches into per-pitcher, per-pitch-type
  grades plus innings and basic results, and write leaderboards.
- **Web app:** serve profiles and leaderboards, and let a user edit pitch types and
  trigger a re-score.

## Data Flow
A user (or a scheduled run) ingests pitch data → the pitches are stored in the raw
tables → feature engineering computes shape features → the model scores each pitch and
writes the scored tables → the profile builder summarizes scored pitches into cards
and leaderboards → the web app serves those results to the user. When a user edits a
pitch's type in the editor, the affected pitches are re-scored and the updated grades
are returned.

## Initial Architecture Sketch
```
                 Public pitch-tracking data
                            |
                       [ Scraper ]
                            v
                   Raw pitch tables (DB)
                            |
                 [ Feature Engineering ]
                            v
                   [ Model / Scoring ] <-- trained model + frozen norms
                            v
                  Scored pitch tables (DB)
                            |
                  [ Profile / Leaderboard Builder ]
                            v
              JSON profiles + leaderboards
                            |
                      [ Flask Web App ] <--> User (profiles, leaderboards, editor)
```

## Open Questions
- How should results be presented for **partially-tracked levels**, where only some
  games have pitch data (so season totals are incomplete)?
- Should re-scoring after a pitch-type edit happen **live in the app** or as a saved
  batch job?
- What is the best **storage layout** as more seasons and levels are added (one
  database vs. several)?
- How automated should the **data refresh** be in future sprints?
