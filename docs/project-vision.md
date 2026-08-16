# PitchGrader — Project Vision

**Grade the nastiness of every pitch, not where it ended up.**

Live app: https://akanemvp.up.railway.app

## Problem & Motivation

Traditional pitching statistics — ERA, K/BB, and the like — conflate a pitcher's actual stuff with everything around it: defense, luck, sequencing, count leverage, and command. A nasty pitch thrown down the middle and hit hard looks bad in the box score; a mediocre pitch that induces a weak groundout looks good. None of these numbers isolate the one thing a pitcher most directly controls and most reliably repeats: the *shape* of the pitch itself.

PitchGrader exists to measure that shape alone. It computes a **Stuff+** grade for every pitch purely from its physical characteristics — velocity, spin rate, movement, release point, and arm slot — independent of where the pitch was located or what the count was. A pitch is graded the same regardless of where it ended up or who was hitting.

The grade is normalized so that **100 = league average** and **every 10 points = one standard deviation**. A 120 fastball sits roughly in the top decile of pitches.

## Target Users

- **Baseball analysts** who want a shape-based, location-independent measure of pitch quality.
- **Player-development staff** evaluating and tracking a pitcher's arsenal.
- **Scouts** assessing MLB pitchers and minor-league prospects.
- **Stats-minded fans** who follow public "Stuff+" / proStuff+ / Pitching Bot–style models.

PitchGrader covers MLB and minor-league feeds — AAA, spring, ACL, FSL, college, and Futures — so prospects and established pitchers can be graded on the same scale.

## Vision & Goals

PitchGrader aims to be a dependable, public-spirited Stuff+ system that:

- Isolates the pitcher-controllable, repeatable quality of a pitch's shape.
- Places every pitch — MLB and minor-league alike — on a single, interpretable grading scale.
- Grades a left-handed and right-handed pitcher throwing identical pitches identically.
- Stays current through the season, re-scoring in-progress games as new data arrives.
- Remains reproducible: grades are stable and bounded across retrains.

## Scope

### In scope

- Scoring **every pitch** in the 2026 MLB season plus the AAA, spring, ACL, FSL, college, and Futures feeds.
- Per-pitcher **cards and leaderboards**: per-pitch-type Stuff+, arsenal breakdowns, and season lines pulled from official MLB boxscores.
- **Live updates** during the season — a background worker re-scrapes and re-scores in-progress games periodically.
- A web-based **pitch-type editor** that lets a user correct mislabeled pitch types; corrections persist and are re-applied on every re-score.

### Out of scope (non-goals)

- **No location or command modeling.** PitchGrader is deliberately shape-only — "stuff" only. It does not judge where a pitch was thrown.
- **No outcome prediction.** It does not forecast game results, player projections, or betting lines.
- **Not a real-time pitch-by-pitch feed.** Updates are periodic, not instantaneous.

## Approach (High Level)

At its core is a machine-learning model — gradient-boosted trees (LightGBM) — trained on roughly **2.9 million pitches from the 2022–2025 seasons**. The model turns pitch shape into an expected run value, then into a z-scored grade on the 100-is-average, 10-points-per-standard-deviation scale.

Pitches are split into three groups — fastballs, breaking balls, and offspeed, with cutters routed by a classifier — and each group is scored by its own trained heads. All groups are then placed on **one shared grading scale** so grades are comparable across pitch types. The model's inputs are **8 arm-normalized shape features**, so handedness does not affect the grade; it uses no location, count, or game-state information. The current season (2026) is scored fully **out-of-sample**.

The system ships as a Python 3 application: LightGBM, pandas, NumPy, and scikit-learn power the model; Flask and gunicorn serve the web app; SQLite stores the data; and it deploys on Railway. A single CLI entry point, `main.py`, exposes `train`, `score`, `profiles`, and `live` subcommands. (Deeper technical detail lives in `architecture.md`.)

## What Success Looks Like

- **The grades match the eye test.** Elite pitches grade highly — top power four-seams and sharp breaking balls lead their groups.
- **Another engineer can reproduce it.** From the README alone, a new engineer can clone, install, train, score, and run the web app.
- **The grades are trustworthy.** They are stable, bounded, and reproducible across retrains.
