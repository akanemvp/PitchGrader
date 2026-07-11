# Stuff+ Pitch Quality Model

## Project Name
**Stuff+** — a pitch-quality model and web app for grading baseball pitchers.

## Overview
Stuff+ scores every tracked pitch a pitcher throws on a single, easy-to-read scale
where **100 is league average and every 10 points is one standard deviation**
(so a 120 fastball is two standard deviations above average). The grade is based
only on the *physical shape* of the pitch — velocity, movement, spin, release point,
and arm slot — not on the result of the at-bat. The project ingests public
pitch-tracking data, engineers shape features, trains a model against frozen
2022–2024 Major League norms, and serves the results through a web app that shows
per-pitcher profiles, leaderboards, and an interactive pitch-type editor.

## Current Status
The project is **functional end-to-end**. It can:
- ingest pitch data for MLB, several minor-league/spring levels, and tracked NCAA college games,
- engineer shape features and score every pitch with the current model, and
- generate per-player profile cards and per-season leaderboards.

**Sprint 2** delivers a working prototype: a Flask web app with an interactive
pitch-type editor that re-classifies and re-scores pitches live (the Sprint 2 feature
slice), built on the Sprint 1 project definition and architecture.

## Project Documents
**Sprint 2**
- [docs/architecture.md](docs/architecture.md) — current system architecture and the Sprint 2 feature slice
- [docs/manual-verification.md](docs/manual-verification.md) — how the prototype is verified by hand
- [docs/ai-implementation-review.md](docs/ai-implementation-review.md) — review of AI-assisted implementation

**Sprint 1**
- [docs/project/project-vision.md](docs/project/project-vision.md) — the problem, users, goals, and scope
- [docs/project/requirements.md](docs/project/requirements.md) — functional, data, and non-functional requirements
- [docs/project/architecture.md](docs/project/architecture.md) — original architecture write-up

## Setup Notes
> Setup is still being finalized and will be expanded as the project stabilizes.

Rough run instructions (Python 3.x):
1. Install dependencies: `pip install -r requirements.txt`
2. Score a season into the database: `python3 main.py score 2025`
3. Build profile cards/leaderboards: `python3 main.py profiles`
4. Run the web app locally: `python3 app.py` (serves on `http://127.0.0.1:5001`)

The app reads a local SQLite database of pitch data. Model artifacts live under
`model/artifacts/`. A full ingestion + training pipeline is run separately and is
not required to view already-scored data.
