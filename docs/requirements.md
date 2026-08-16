# PitchGrader — Requirements

This document specifies the functional and non-functional requirements for PitchGrader, a baseball analytics web application that computes a shape-based **Stuff+** grade for every pitch. Grades are normalized so that 100 = league average and every 10 points = one standard deviation.

## 1. Functional Requirements

### Scoring

- **FR-1** — The system shall compute a Stuff+ grade for every pitch in the 2026 MLB season and in the AAA, spring, ACL, FSL, college, and Futures feeds.
- **FR-2** — The system shall grade each pitch purely from its physical shape (velocity, spin rate, movement, release point, arm slot), using no pitch location, count, or game-state information.
- **FR-3** — The system shall place all pitches on a single shared grading scale where 100 = league average and each 10 points = one standard deviation, so grades are comparable across pitch types.
- **FR-4** — The system shall route each pitch into one of three groups — fastballs, breaking balls, or offspeed — with cutters assigned by a classifier, and score each group with its own trained model heads.

### Cards & leaderboards

- **FR-5** — The system shall build per-pitcher cards showing per-pitch-type Stuff+ and an arsenal breakdown.
- **FR-6** — The system shall present season pitching lines on pitcher cards, pulled from official MLB boxscores.
- **FR-7** — The system shall build leaderboards ranking pitchers (and pitch types) by Stuff+.

### Live updates

- **FR-8** — The system shall run a background worker that periodically re-scrapes and re-scores in-progress games during the season.
- **FR-9** — The system shall update pitcher cards and leaderboards to reflect newly scored pitches from re-scored games.

### Pitch-type editor

- **FR-10** — The system shall provide a web-based pitch-type editor that lets a user correct a mislabeled pitch type.
- **FR-11** — The system shall persist user pitch-type corrections and re-apply them automatically on every subsequent re-score.

### CLI & pipeline

- **FR-12** — The system shall expose a CLI entry point (`main.py`) with `train`, `score`, `profiles`, and `live` subcommands.
- **FR-13** — The system shall, via `train`, train the model on the historical pitch dataset (2022–2025 seasons).
- **FR-14** — The system shall, via `score`, apply the trained model to produce Stuff+ grades for pitches.
- **FR-15** — The system shall, via `profiles`, generate the per-pitcher cards and leaderboards.
- **FR-16** — The system shall, via `live`, run the live re-scrape and re-score worker.

### Web UI

- **FR-17** — The system shall serve a web application (Flask + gunicorn) through which users can browse pitcher cards, leaderboards, and the pitch-type editor.

## 2. Non-Functional Requirements

- **NFR-1 (Reproducibility / determinism)** — Grades shall be stable, bounded, and reproducible across retrains; the same inputs shall yield the same grades.
- **NFR-2 (Handedness invariance)** — The model shall use 8 arm-normalized shape features so that a left-handed and a right-handed pitcher throwing identical pitches receive identical grades.
- **NFR-3 (Shape-only constraint)** — The model shall never consume pitch location, count, or game state; grading depends on pitch shape alone.
- **NFR-4 (Out-of-sample scoring)** — The current season (2026) shall be scored fully out-of-sample, i.e., its pitches are not part of the training data.
- **NFR-5 (Performance / scale)** — The system shall score hundreds of thousands of pitches per season across all supported feeds.
- **NFR-6 (Runnability from README)** — Another engineer shall be able to clone, install, train, score, and run the web app using only the instructions in the README.
- **NFR-7 (Test coverage)** — The system shall include automated tests covering its core scoring and pipeline behavior.
- **NFR-8 (Deployability)** — The system shall be deployable on Railway and run there as a Flask + gunicorn service.
- **NFR-9 (Eye-test validity)** — Grades shall align with the eye test: elite pitches (e.g., top power four-seams, sharp breaking balls) shall lead their groups.

## 3. Constraints & Assumptions

- **C-1** — Scoring a pitch requires its full 9-parameter trajectory (release velocity and acceleration vectors, extension, arm slot, release point); the induced-movement features are derived from kinematics, so a pitch is graded without needing Statcast `spin_axis`. A pitch missing any shape feature is left ungraded.
- **C-2** — The model is trained on roughly 2.9 million pitches from the 2022–2025 seasons.
- **C-3** — Training and scoring data are stored in a SQLite database that is git-ignored locally and mounted as a volume on Railway.
- **C-4** — The technology stack is fixed: Python 3, LightGBM, pandas, NumPy, and scikit-learn for the model; Flask + gunicorn for the web app; SQLite for storage; Railway for deployment.
